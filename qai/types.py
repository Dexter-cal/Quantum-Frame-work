"""
qai.types -- type aliases + Protocol definitions + error handling helpers.

Production hardening pass:
  - Type aliases for common qai types
  - Protocol classes (PEP 544) for duck-typing contracts
  - Custom exception hierarchy
  - Error-handling decorators (retry, safe_call, validate_input)
  - Result type (Rust-style Ok/Err) for safer error propagation
"""
from __future__ import annotations
import functools
import time
import traceback
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar,
    Union, Protocol, runtime_checkable,
)
import numpy as np


# ============================================================================
# Type aliases
# ============================================================================
ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...], "torch.Tensor"]
TensorLike = Union[np.ndarray, "torch.Tensor"]
ModelLike = Any  # anything with .forward / .fit
LLMCallable = Callable[[str], str]
EmbedderLike = Callable[[str], np.ndarray]
RetrieverLike = Callable[[str, int], List[Tuple[str, float]]]

T = TypeVar("T")
E = TypeVar("E")


# ============================================================================
# Protocols (PEP 544) — duck-typed contracts
# ============================================================================
@runtime_checkable
class Forwardable(Protocol):
    """Anything with a forward() method."""
    def forward(self, x: Any) -> Any: ...


@runtime_checkable
class Fittable(Protocol):
    """Anything with a fit() method."""
    def fit(self, X: Any, y: Any = None) -> Any: ...


@runtime_checkable
class Samplable(Protocol):
    """Anything with a sample() method (e.g. generative models)."""
    def sample(self, n: int) -> Any: ...


@runtime_checkable
class Stateful(Protocol):
    """Anything with state_dict() / load_state_dict()."""
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, sd: Dict[str, Any]) -> None: ...


@runtime_checkable
class TokenizerLike(Protocol):
    """Tokenizer protocol."""
    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...


# ============================================================================
# Exception hierarchy
# ============================================================================
class QAIError(Exception):
    """Base class for all qai errors."""
    pass


class TechniqueNotFoundError(QAIError):
    """Raised when qai.build() is called with an unknown technique."""
    pass


class ModelNotTrainedError(QAIError):
    """Raised when forward() is called on an untrained model."""
    pass


class BundleCorruptError(QAIError):
    """Raised when a .qmodel bundle is corrupt or tampered."""
    pass


class BundleSignatureError(QAIError):
    """Raised when a .qmodel bundle's signature doesn't match."""
    pass


class GPUUnavailableError(QAIError):
    """Raised when a GPU operation is requested but no GPU is available."""
    pass


class OutOfMemoryError(QAIError):
    """Raised when an operation runs out of RAM or VRAM."""
    pass


class ConvergenceError(QAIError):
    """Raised when training fails to converge."""
    pass


class DataValidationError(QAIError):
    """Raised when input data fails validation."""
    pass


class ConfigurationError(QAIError):
    """Raised when a configuration parameter is invalid."""
    pass


# ============================================================================
# Result type (Rust-style Ok/Err)
# ============================================================================
class Ok(Generic[T]):
    """Success result."""
    def __init__(self, value: T):
        self.value = value
        self.is_ok = True
        self.is_err = False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def map(self, fn: Callable[[T], Any]) -> "Ok":
        return Ok(fn(self.value))


class Err(Generic[E]):
    """Failure result."""
    def __init__(self, error: E):
        self.error = error
        self.is_ok = False
        self.is_err = True

    def unwrap(self) -> Any:
        raise self.error if isinstance(self.error, Exception) else QAIError(str(self.error))

    def unwrap_or(self, default: T) -> T:
        return default

    def map(self, fn: Callable[[T], Any]) -> "Err":
        return self


Result = Union[Ok, Err]


def ok(value: Any) -> Ok:
    return Ok(value)


def err(error: Any) -> Err:
    return Err(error)


# ============================================================================
# Error-handling decorators
# ============================================================================
def retry(max_attempts: int = 3, delay: float = 1.0,
           exceptions: Tuple[Type[Exception], ...] = (Exception,)) -> Callable:
    """Retry a function on failure.

    @retry(max_attempts=3, delay=1.0)
    def flaky_operation(): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


def safe_call(default: Any = None,
               exceptions: Tuple[Type[Exception], ...] = (Exception,)) -> Callable:
    """Catch exceptions and return a default value.

    @safe_call(default=None)
    def risky_operation(): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except exceptions:
                return default
        return wrapper
    return decorator


def validate_input(validator: Callable[[Any], bool],
                    error_msg: str = "invalid input") -> Callable:
    """Validate the first argument before calling the function.

    @validate_input(lambda x: x is not None, "input must not be None")
    def process(x): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if args and not validator(args[0]):
                raise DataValidationError(error_msg)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def timed(log_fn: Optional[Callable[[str, float], None]] = None) -> Callable:
    """Time a function and log the result.

    @timed()
    def slow_function(): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - t0) * 1000
                if log_fn:
                    log_fn(fn.__name__, elapsed)
                else:
                    print(f"[timed] {fn.__name__}: {elapsed:.2f}ms")
        return wrapper
    return decorator


def memoize(maxsize: int = 128) -> Callable:
    """Memoize a function's results (LRU cache)."""
    def decorator(fn: Callable) -> Callable:
        cache: Dict[Tuple, Any] = {}
        order: List[Tuple] = []
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                return cache[key]
            result = fn(*args, **kwargs)
            cache[key] = result
            order.append(key)
            if len(cache) > maxsize:
                oldest = order.pop(0)
                cache.pop(oldest, None)
            return result
        wrapper.cache_clear = lambda: (cache.clear(), order.clear())
        return wrapper
    return decorator


def deprecated(replacement: Optional[str] = None) -> Callable:
    """Mark a function as deprecated; warn on use."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            import warnings
            msg = f"{fn.__name__} is deprecated."
            if replacement:
                msg += f" Use {replacement} instead."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Input validation helpers
# ============================================================================
def require(condition: bool, message: str,
             exception: Type[Exception] = ValueError) -> None:
    """Raise an exception if condition is False."""
    if not condition:
        raise exception(message)


def require_not_none(value: Any, name: str = "value") -> None:
    require(value is not None, f"{name} must not be None")


def require_type(value: Any, expected: Type, name: str = "value") -> None:
    require(isinstance(value, expected),
             f"{name} must be {expected.__name__}, got {type(value).__name__}")


def require_shape(arr: np.ndarray, expected: Tuple[int, ...],
                   name: str = "array") -> None:
    require(arr.shape == expected,
             f"{name} must have shape {expected}, got {arr.shape}")


def require_in_range(value: float, lo: float, hi: float,
                       name: str = "value") -> None:
    require(lo <= value <= hi,
             f"{name} must be in [{lo}, {hi}], got {value}")


# ============================================================================
# Logging helper
# ============================================================================
class Logger:
    """Minimal logger that doesn't require the stdlib logging module."""
    LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "FATAL": 50}

    def __init__(self, name: str = "qai", level: str = "INFO"):
        self.name = name
        self.level = self.LEVELS.get(level, 20)
        self._handlers: List[Callable[[str, str], None]] = []

    def add_handler(self, fn: Callable[[str, str], None]) -> None:
        self._handlers.append(fn)

    def log(self, level: str, msg: str) -> None:
        if self.LEVELS.get(level, 20) < self.level:
            return
        line = f"[{level}] {self.name}: {msg}"
        for h in self._handlers:
            try:
                h(level, line)
            except Exception:
                pass
        if not self._handlers:
            print(line)

    def debug(self, msg: str): self.log("DEBUG", msg)
    def info(self, msg: str): self.log("INFO", msg)
    def warn(self, msg: str): self.log("WARN", msg)
    def error(self, msg: str): self.log("ERROR", msg)
    def fatal(self, msg: str): self.log("FATAL", msg)


__all__ = [
    # Type aliases
    "ArrayLike", "TensorLike", "ModelLike", "LLMCallable",
    "EmbedderLike", "RetrieverLike",
    # Protocols
    "Forwardable", "Fittable", "Samplable", "Stateful", "TokenizerLike",
    # Exceptions
    "QAIError", "TechniqueNotFoundError", "ModelNotTrainedError",
    "BundleCorruptError", "BundleSignatureError", "GPUUnavailableError",
    "OutOfMemoryError", "ConvergenceError", "DataValidationError",
    "ConfigurationError",
    # Result type
    "Ok", "Err", "Result", "ok", "err",
    # Decorators
    "retry", "safe_call", "validate_input", "timed", "memoize", "deprecated",
    # Validation helpers
    "require", "require_not_none", "require_type", "require_shape",
    "require_in_range",
    # Logger
    "Logger",
]
