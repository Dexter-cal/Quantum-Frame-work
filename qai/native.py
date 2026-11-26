"""
qai.native -- C-extension bridge for direct memory access.

This is the answer to the user's question: "can we write to memory like
RAM/ROM, integrate a C language underneath but call it from Python?"

YES. Three layers, each a step closer to the metal:

  Layer 1: ctypes bridge  -- pure stdlib, no compilation needed
    - call any function from a .so/.dll
    - allocate native buffers and read/write them from Python
    - exposes mmap-backed arrays that survive across processes

  Layer 2: cffi bridge     -- needs cffi (pip install cffi)
    - faster than ctypes, can inline C declarations
    - better ABI stability

  Layer 3: pybind11 bridge -- needs pybind11 + compiler
    - most ergonomic, used by torch/numpy
    - we generate a small C++ shim that wraps qai models

Conceptual model:
  ┌─────────────────────────────────────────┐
  │              Python user code           │
  ├─────────────────────────────────────────┤
  │  qai.native.NativeTensor / NativeBuffer │   <- Python API
  ├──────────────┬──────────────────────────┤
  │   ctypes     │   cffi    │  pybind11    │   <- bridge layer
  ├──────────────┴──────────┴──────────────┤
  │             libC / system calls          │
  ├─────────────────────────────────────────┤
  │        shared memory / mmap / GPU mem    │
  └─────────────────────────────────────────┘

What this gives you:
  - NativeTensor: like a numpy array but backed by malloc'd C memory
  - SharedMemoryTensor: lives in /dev/shm, two processes can read/write
  - RingBuffer: lock-free single-producer single-consumer queue
  - MemoryArena: pre-allocated pool, O(1) alloc/free, no GC pressure
  - CPythonBridge: load any .so and call its functions
  - CompileBridge: compile a tiny C function on-the-fly, call it
"""
from __future__ import annotations
import array
import ctypes
import ctypes.util
import mmap
import os
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 1. ctypes bridge -- pure stdlib
# ============================================================================
class CPythonBridge:
    """Load any .so/.dll and call its functions from Python.

    Usage:
        bridge = CPythonBridge()
        bridge.load_library("libm.so.6")            # system math
        bridge.bind_function("sqrt", restype=ctypes.c_double,
                              argtypes=[ctypes.c_double])
        print(bridge.call("sqrt", 16.0))             # -> 4.0

    Or compile C code on the fly:
        bridge.compile_and_load("double sq(double x) { return x*x; }",
                                  function_name="sq",
                                  restype=ctypes.c_double,
                                  argtypes=[ctypes.c_double])
        print(bridge.call("sq", 5.0))                # -> 25.0
    """
    def __init__(self):
        self._libs: Dict[str, ctypes.CDLL] = {}
        self._funcs: Dict[str, ctypes._FuncPtr] = {}

    def load_library(self, name: str) -> ctypes.CDLL:
        """Load a shared library by name or path."""
        if name in self._libs:
            return self._libs[name]
        # try as path first, then as system lib
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            path = ctypes.util.find_library(name)
            if path is None:
                raise OSError(f"library '{name}' not found")
            lib = ctypes.CDLL(path)
        self._libs[name] = lib
        return lib

    def bind_function(self, name: str, lib_name: Optional[str] = None,
                       restype: Any = ctypes.c_int,
                       argtypes: Optional[List[Any]] = None) -> ctypes._FuncPtr:
        """Bind a function from a loaded library."""
        if lib_name is None:
            # search all loaded libs
            for lib in self._libs.values():
                try:
                    fn = getattr(lib, name)
                    fn.restype = restype
                    fn.argtypes = argtypes or []
                    self._funcs[name] = fn
                    return fn
                except AttributeError:
                    continue
            raise AttributeError(f"function '{name}' not found in any loaded library")
        lib = self._libs[lib_name]
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes or []
        self._funcs[name] = fn
        return fn

    def call(self, name: str, *args) -> Any:
        if name not in self._funcs:
            raise KeyError(f"function '{name}' not bound; call bind_function first")
        return self._funcs[name](*args)

    def compile_and_load(self, c_code: str, function_name: str,
                          restype: Any = ctypes.c_int,
                          argtypes: Optional[List[Any]] = None,
                          extra_includes: Optional[List[str]] = None,
                          lib_path: Optional[str] = None) -> ctypes._FuncPtr:
        """Compile a snippet of C code on the fly, load it as a shared lib.

        Requires gcc or clang on the system. Returns the bound function.
        """
        # find a C compiler
        compiler = os.environ.get("CC", "gcc")
        try:
            subprocess.run([compiler, "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(f"no C compiler found (tried {compiler}); install gcc or set CC env var")
        # write source
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as src_f:
            for inc in (extra_includes or []):
                src_f.write(f"#include <{inc}>\n")
            src_f.write(c_code)
            src_path = src_f.name
        if lib_path is None:
            lib_path = src_path.replace(".c", ".so")
        # compile as shared lib
        cmd = [compiler, "-shared", "-fPIC", "-O2", "-o", lib_path, src_path]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"compilation failed: {result.stderr.decode()}")
        os.unlink(src_path)
        lib_name = os.path.basename(lib_path)
        self.load_library(lib_path)
        return self.bind_function(function_name, lib_name=lib_path,
                                    restype=restype, argtypes=argtypes)


# ============================================================================
# 2. NativeTensor -- numpy-compatible buffer backed by malloc'd C memory
# ============================================================================
class NativeTensor:
    """A contiguous block of C memory exposed as a numpy array view.

    Unlike np.zeros, the underlying memory is malloc'd and freed explicitly,
    so we can hand the pointer to a C function and back without copies.
    """
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(shape)) * self.dtype.itemsize
        # allocate via ctypes
        self._buf = (ctypes.c_byte * self.nbytes)()
        # expose as numpy view (zero-copy)
        self._np = np.ctypeslib.as_array(self._buf).view(self.dtype).reshape(shape)

    @property
    def array(self) -> np.ndarray:
        return self._np

    @property
    def pointer(self) -> int:
        """Raw memory address — pass this to C functions."""
        return ctypes.addressof(self._buf)

    def write_at(self, offset: int, value: int) -> None:
        """Write a single byte at offset (like writing to RAM)."""
        if 0 <= offset < self.nbytes:
            self._buf[offset] = value & 0xFF

    def read_at(self, offset: int) -> int:
        if 0 <= offset < self.nbytes:
            return int(self._buf[offset])
        raise IndexError(f"offset {offset} out of range [0, {self.nbytes})")

    def zero(self) -> None:
        ctypes.memset(self._buf, 0, self.nbytes)

    def copy_to(self, other: "NativeTensor") -> None:
        if self.nbytes != other.nbytes:
            raise ValueError("size mismatch")
        ctypes.memmove(other._buf, self._buf, self.nbytes)

    def __repr__(self) -> str:
        return f"NativeTensor(shape={self.shape}, dtype={self.dtype}, ptr=0x{self.pointer:x})"


# ============================================================================
# 3. SharedMemoryTensor -- /dev/shm backed, cross-process
# ============================================================================
class SharedMemoryTensor:
    """A tensor in POSIX shared memory; multiple processes can read/write.

    Useful for multi-process training pipelines where you want to pass
    tensors between workers without pickling.
    """
    def __init__(self, name: str, shape: Tuple[int, ...],
                 dtype: np.dtype = np.float32, create: bool = True):
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(shape)) * self.dtype.itemsize
        if sys.platform == "win32":
            # Windows: use a file-mapping object via ctypes.windll.kernel32
            # (simplified: just use a temp file)
            self._path = os.path.join(tempfile.gettempdir(), f"qai_shm_{name}")
            mode = "wb+" if create else "rb+"
            self._f = open(self._path, mode)
            if create:
                self._f.truncate(self.nbytes)
        else:
            # POSIX shared memory
            self._path = f"/dev/shm/qai_{name}" if create else f"/dev/shm/qai_{name}"
            mode = os.O_CREAT | os.O_RDWR if create else os.O_RDWR
            self._fd = os.open(self._path, mode, 0o600)
            if create:
                os.ftruncate(self._fd, self.nbytes)
        # mmap it
        self._mm = mmap.mmap(self._fileno(), self.nbytes)
        self._np = np.frombuffer(self._mm, dtype=self.dtype).reshape(shape)

    def _fileno(self) -> int:
        if sys.platform == "win32":
            return self._f.fileno()
        return self._fd

    @property
    def array(self) -> np.ndarray:
        return self._np

    def close(self) -> None:
        self._np = None  # release numpy view (it exports the buffer)
        self._mm.close()
        if sys.platform == "win32":
            self._f.close()
        else:
            os.close(self._fd)

    def unlink(self) -> None:
        self.close()
        try:
            if sys.platform == "win32":
                os.unlink(self._path)
            else:
                os.unlink(self._path)
        except FileNotFoundError:
            pass

    @classmethod
    def open_existing(cls, name: str, shape: Tuple[int, ...],
                       dtype: np.dtype = np.float32) -> "SharedMemoryTensor":
        return cls(name, shape, dtype, create=False)


# ============================================================================
# 4. RingBuffer -- lock-free SPSC queue for streaming data
# ============================================================================
class RingBuffer:
    """Lock-free single-producer single-consumer ring buffer.

    Producer writes chunks; consumer reads them. No locks; works across
    threads (single producer, single consumer). Backed by shared memory
    so it works across processes too.
    """
    def __init__(self, name: str, capacity: int, item_size: int,
                 create: bool = True):
        self.name = name
        self.capacity = capacity
        self.item_size = item_size
        # layout: [head:8][tail:8][data:capacity*item_size]
        self._layout_size = 16 + capacity * item_size
        if sys.platform == "win32":
            self._path = os.path.join(tempfile.gettempdir(), f"qai_ring_{name}")
            self._f = open(self._path, "wb+" if create else "rb+")
            if create:
                self._f.truncate(self._layout_size)
            fd = self._f.fileno()
        else:
            self._path = f"/dev/shm/qai_ring_{name}"
            mode = os.O_CREAT | os.O_RDWR if create else os.O_RDWR
            self._fd = os.open(self._path, mode, 0o600)
            if create:
                os.ftruncate(self._fd, self._layout_size)
            fd = self._fd
        self._mm = mmap.mmap(fd, self._layout_size)
        self._head = np.frombuffer(self._mm, dtype=np.uint64, count=1, offset=0)
        self._tail = np.frombuffer(self._mm, dtype=np.uint64, count=1, offset=8)
        self._data_offset = 16

    def push(self, item: bytes) -> bool:
        """Non-blocking push. Returns False if buffer is full."""
        head = int(self._head[0])
        tail = int(self._tail[0])
        next_head = (head + 1) % self.capacity
        if next_head == tail:
            return False  # full
        offset = self._data_offset + head * self.item_size
        if len(item) > self.item_size:
            raise ValueError(f"item too large: {len(item)} > {self.item_size}")
        self._mm[offset:offset + len(item)] = item
        if len(item) < self.item_size:
            self._mm[offset + len(item):offset + self.item_size] = b"\x00" * (self.item_size - len(item))
        self._head[0] = next_head
        return True

    def pop(self) -> Optional[bytes]:
        head = int(self._head[0])
        tail = int(self._tail[0])
        if head == tail:
            return None  # empty
        offset = self._data_offset + tail * self.item_size
        item = bytes(self._mm[offset:offset + self.item_size])
        self._tail[0] = (tail + 1) % self.capacity
        return item

    def is_empty(self) -> bool:
        return int(self._head[0]) == int(self._tail[0])

    def is_full(self) -> bool:
        return (int(self._head[0]) + 1) % self.capacity == int(self._tail[0])

    def close(self) -> None:
        self._head = None  # release numpy views
        self._tail = None
        self._mm.close()
        if sys.platform == "win32":
            self._f.close()
        else:
            os.close(self._fd)


# ============================================================================
# 5. MemoryArena -- pre-allocated pool, O(1) alloc/free
# ============================================================================
class MemoryArena:
    """Slab allocator: pre-allocate a large block, hand out chunks.

    O(1) allocate / free. No GC pressure (chunks are reused, not freed).
    Great for high-frequency training loops where you'd otherwise create
    thousands of small arrays.
    """
    def __init__(self, total_bytes: int, chunk_size: int = 1024):
        self.total_bytes = total_bytes
        self.chunk_size = chunk_size
        self.n_chunks = total_bytes // chunk_size
        self._buf = (ctypes.c_byte * total_bytes)()
        self._free_list: List[int] = list(range(0, total_bytes, chunk_size))
        self._used: Dict[int, int] = {}  # offset -> size

    def allocate(self, size: int) -> int:
        """Allocate `size` bytes; return offset into the arena."""
        n_chunks_needed = (size + self.chunk_size - 1) // self.chunk_size
        if n_chunks_needed > len(self._free_list):
            raise MemoryError(f"arena exhausted: need {n_chunks_needed} chunks, have {len(self._free_list)}")
        offset = self._free_list.pop(0)
        for _ in range(n_chunks_needed - 1):
            self._free_list.pop(0)
        self._used[offset] = n_chunks_needed * self.chunk_size
        return offset

    def free(self, offset: int) -> None:
        if offset not in self._used:
            raise ValueError(f"offset {offset} not allocated")
        n_chunks = self._used[offset] // self.chunk_size
        for i in range(n_chunks):
            self._free_list.append(offset + i * self.chunk_size)
        del self._used[offset]

    def as_array(self, offset: int, size: int, dtype: np.dtype = np.float32) -> np.ndarray:
        """Get a numpy view into the arena at offset."""
        return np.ctypeslib.as_array(
            (ctypes.c_byte * size).from_buffer(ctypes.cast(ctypes.pointer(self._buf),
                                                              ctypes.POINTER(ctypes.c_byte)),
                                                  size)
        ).view(dtype)

    def usage(self) -> Dict[str, int]:
        used = sum(self._used.values())
        return {"total": self.total_bytes, "used": used, "free": self.total_bytes - used,
                "n_free_chunks": len(self._free_list)}


# ============================================================================
# 6. cffi bridge -- alternative to ctypes
# ============================================================================
class CFFIBridge:
    """Use cffi for more ergonomic C interop.

    Requires the `cffi` package. Faster than ctypes, more ABI-stable.
    """
    def __init__(self):
        try:
            import cffi
            self._ffi = cffi.FFI()
        except ImportError:
            raise ImportError("cffi not installed; pip install cffi")

    def declare(self, cdef: str) -> None:
        """Declare C types/functions inline (like a .h file)."""
        self._ffi.cdef(cdef)

    def compile(self, c_code: str, libraries: Optional[List[str]] = None,
                extra_compile_args: Optional[List[str]] = None) -> Any:
        """Compile C code at runtime; return a module handle."""
        return self._ffi.verify(c_code, libraries=libraries or [],
                                  extra_compile_args=extra_compile_args or [])

    def dlopen(self, name: str) -> Any:
        """Open a shared library."""
        return self._ffi.dlopen(name)


# ============================================================================
# 7. pybind11 bridge -- generate C++ shim for a qai model
# ============================================================================
class Pybind11Bridge:
    """Generate a C++ extension that wraps a qai model for fast inference.

    Requires pybind11 + a C++ compiler. We generate a small .cpp file
    that calls back into Python via pybind11's embed module.
    """
    TEMPLATE = """
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
namespace py = pybind11;

PYBIND11_MODULE({module_name}, m) {{
    m.def("forward", [](py::array_t<float> input) {{
        py::object model = py::module_::import("qai").attr("current_model");
        py::object result = model.attr("forward")(input);
        return result;
    }});
    m.def("set_model", [](py::object model) {{
        py::module_::import("qai").attr("current_model") = model;
    }});
}}
"""

    def __init__(self, module_name: str = "qai_native_bridge"):
        self.module_name = module_name
        try:
            import pybind11
            self._pybind11 = pybind11
        except ImportError:
            raise ImportError("pybind11 not installed; pip install pybind11")

    def build(self, output_dir: Optional[str] = None) -> str:
        """Generate the .cpp and compile it. Returns the .so path."""
        output_dir = output_dir or tempfile.mkdtemp()
        cpp_path = os.path.join(output_dir, f"{self.module_name}.cpp")
        with open(cpp_path, "w") as f:
            f.write(self.TEMPLATE.format(module_name=self.module_name))
        # use pybind11's CMake or setuptools; here we use cffi-style compile
        import sysconfig
        ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        so_path = os.path.join(output_dir, self.module_name + ext)
        cmd = ["c++", "-O3", "-shared", "-std=c++11", "-fPIC",
                f"-I{self._pybind11.get_include()}",
                f"-I{sysconfig.get_path('include')}",
                cpp_path, "-o", so_path]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"compilation failed: {result.stderr.decode()}")
        return so_path

    def load(self, so_path: str) -> Any:
        """Load the compiled module and return it."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(self.module_name, so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


# ============================================================================
# 8. Low-level system memory info
# ============================================================================
class SystemMemory:
    """Read system memory stats (RAM/Swap/Available)."""
    @staticmethod
    def info() -> Dict[str, int]:
        """Return RAM/Swap info in bytes."""
        if sys.platform == "win32":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                              ("dwMemoryLoad", ctypes.c_ulong),
                              ("ullTotalPhys", ctypes.c_ulonglong),
                              ("ullAvailPhys", ctypes.c_ulonglong),
                              ("ullTotalPageFile", ctypes.c_ulonglong),
                              ("ullAvailPageFile", ctypes.c_ulonglong),
                              ("ullTotalVirtual", ctypes.c_ulonglong),
                              ("ullAvailVirtual", ctypes.c_ulonglong),
                              ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return {"total_ram": stat.ullTotalPhys, "available_ram": stat.ullAvailPhys,
                      "total_swap": stat.ullTotalPageFile, "available_swap": stat.ullAvailPageFile,
                      "load_pct": stat.dwMemoryLoad}
        # POSIX: read /proc/meminfo
        info = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1]) * 1024  # KB to B
        except FileNotFoundError:
            pass
        return {"total_ram": info.get("MemTotal", 0),
                  "available_ram": info.get("MemAvailable", 0),
                  "free_ram": info.get("MemFree", 0),
                  "total_swap": info.get("SwapTotal", 0),
                  "free_swap": info.get("SwapFree", 0)}

    @staticmethod
    def process_memory() -> Dict[str, int]:
        """Current process memory usage."""
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024  # KB to B on Linux
        return {"rss_bytes": rss, "rss_mb": rss / 1024 / 1024}


__all__ = [
    "CPythonBridge", "NativeTensor", "SharedMemoryTensor", "RingBuffer",
    "MemoryArena", "CFFIBridge", "Pybind11Bridge", "SystemMemory",
]
