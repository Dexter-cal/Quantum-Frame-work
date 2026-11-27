"""
qai._torch_stub -- graceful torch fallback so `import qai` works without torch.

PyTorch is an *optional* dependency of QAI. The neural-network, transformer,
and CNN techniques inherit their internal building blocks from ``torch.nn.Module``
at class-definition time. If torch is missing, those ``class Foo(nn.Module)``
lines raise ``NameError`` and the whole package fails to import — even though
the user may only want a numpy-only technique such as ``classifier`` or
``time_series``.

This module provides stand-ins that let the class definitions succeed. The
stubs raise a clear ``ImportError`` (via :func:`require_torch`) the moment a
stub-backed class is *instantiated*, so the failure mode is the same as
before — just deferred from import time to use time.

Usage in a technique module::

    from qai._torch_stub import try_import_torch
    torch, nn, F, _TORCH_AVAILABLE = try_import_torch()

    def _require_torch():
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "[WHAT] this technique requires PyTorch\\n"
                "[WHY] it uses torch.nn for its internals\\n"
                "[FIX] pip install torch"
            )

    if _TORCH_AVAILABLE:
        class _ConvBlock(nn.Module):
            ...
    else:
        # Stub base — subclasses import fine, instantiation raises
        from qai._torch_stub import StubModule as nn_Module  # noqa
        class _ConvBlock(nn_Module):
            ...
"""
from __future__ import annotations
from typing import Any


class StubModule:
    """Stand-in base class for ``torch.nn.Module`` when torch is missing.

    Subclassing this at module load time is always safe — no metaclass magic,
    no ``__init_subclass__`` side effects. *Instantiating* a subclass raises
    ``ImportError`` so callers get a clear "install torch" message instead of
    a confusing ``NameError`` or ``AttributeError``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "[WHAT] this technique requires PyTorch to instantiate\n"
            "[WHY] the class inherits from torch.nn.Module, but torch is "
            "not installed\n"
            "[FIX] install torch: pip install torch"
        )


class _NNStubNamespace:
    """Object whose every attribute access returns :class:`StubModule`.

    Lets ``class Foo(nn.Module)`` and ``class Bar(nn.Sequential)`` style
    definitions succeed at import time when torch is absent. Only attribute
    *access* is stubbed — calling ``nn.ReLU()`` would still raise via
    ``StubModule.__init__``.
    """

    def __getattr__(self, name: str) -> type:
        return StubModule


def _identity_decorator(*args: Any, **kwargs: Any):
    """Decorator/context-manager that does nothing.

    Used as a stand-in for ``torch.no_grad()``, ``torch.enable_grad()``,
    ``torch.inference_mode()`` etc. when torch is absent. Supports three
    usage patterns:

    - ``@torch.no_grad`` (bare decorator, no parens): ``args[0]`` is the
      function. Return it unchanged.
    - ``@torch.no_grad()`` (called as decorator factory): returns an object
      that is *both* callable (so it can wrap the function) AND a context
      manager (so ``with torch.no_grad():`` also works).
    - ``with torch.no_grad():`` (context manager): same returned object.
    """
    # Case 1: `@torch.no_grad` (bare) — args[0] is the function
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    # Case 2 & 3: `torch.no_grad()` returns an object that is simultaneously
    # a decorator (callable) and a context manager.
    class _CMD:
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _CMD()


class _TorchStubNamespace:
    """Stand-in for the ``torch`` module when torch is not installed.

    Attribute access returns :func:`_identity_decorator` by default, so
    decorators like ``@torch.no_grad()`` work at class-definition time.
    The ``nn`` and ``nn.functional`` aliases are :class:`_NNStubNamespace`
    so ``class Foo(nn.Module)`` style definitions succeed too.
    """

    @property
    def nn(self):
        return _NNStubNamespace()

    @property
    def nn_functional(self):
        return _NNStubNamespace()

    def __getattr__(self, name: str):
        # Default: return identity decorator (handles no_grad, enable_grad,
        # inference_mode, autocast, etc.)
        return _identity_decorator


def try_import_torch():
    """Try to import torch, nn, F. Return stubs on ImportError.

    Returns
    -------
    tuple
        ``(torch, nn, F, available)`` where the first three are either the
        real modules or stubs, and ``available`` is a bool.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F, True
    except ImportError:
        torch_stub = _TorchStubNamespace()
        nn_stub = _NNStubNamespace()
        return torch_stub, nn_stub, nn_stub, False


__all__ = ["StubModule", "try_import_torch"]
