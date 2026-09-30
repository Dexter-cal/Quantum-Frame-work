"""
qai.core.logic -- the Python equivalent of the design doc's `logic {}` block
(Section 74/84).

DESIGN DECISION (made here, not improvised mid-code):
Quantum's logic{} requires an explicit input() at entry and output() at
exit -- nothing implicit, nothing assumed. Python already has an exact
structural equivalent: a FUNCTION's signature IS its declared input, and
`return` IS its explicit output. So `logic {}` becomes a decorator that
enforces that same discipline using Python's own mechanisms, instead of
inventing new syntax Python doesn't have:

    @qai.logic
    def my_decision(x):
        result = model.predict(x)
        return result          # <- this IS output(), mandatory, checked

This gives us, for free, from Python itself:
- input()  -> the function's own parameter(s)
- output() -> the function's own return statement
- entry/exit tracking -> the decorator wraps around the call
- the "must have an exit" rule (Section 84) -> enforced by checking the
  function actually returned something, not None by accident
"""
from __future__ import annotations
import time
import functools


class LogicError(RuntimeError):
    """Raised when a logic function violates the input/output contract."""


def logic(fn=None, *, name=None, max_seconds=None):
    """Decorator implementing the logic{} contract.

    Usage:
        @qai.logic
        def my_logic(x):
            ...
            return result

        @qai.logic(name="retry_until_confident", max_seconds=5)
        def my_logic(x):
            ...
            return result
    """
    def decorator(func):
        logic_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()

            result = func(*args, **kwargs)

            elapsed = time.time() - start

            # ENFORCE: output() is mandatory -- a logic block that falls off
            # the end without returning anything is exactly the "loop that
            # never reaches output()" danger named in Section 84/95. Catch
            # it here, structurally, rather than trusting every author.
            if result is None:
                raise LogicError(
                    f"[WHAT] logic function '{logic_name}' completed without returning a value\n"
                    f"[WHY] Every logic block must have an explicit output -- returning None "
                    f"is treated as a missing exit, not a valid (empty) result\n"
                    f"[FIX] Add an explicit 'return <value>' at the end of '{logic_name}'"
                )

            if max_seconds is not None and elapsed > max_seconds:
                raise LogicError(
                    f"[WHAT] logic function '{logic_name}' took {elapsed:.2f}s\n"
                    f"[WHY] max_seconds={max_seconds} was set as a hard time budget\n"
                    f"[FIX] Either optimize '{logic_name}' or raise max_seconds if this is expected"
                )

            wrapper.last_run = {"logic_name": logic_name, "elapsed_s": round(elapsed, 4), "result_type": type(result).__name__}
            return result

        wrapper.logic_name = logic_name
        wrapper.last_run = None
        return wrapper

    if fn is not None:
        # used as @qai.logic with no parens
        return decorator(fn)
    return decorator


def save_logic_source(logic_fn, path):
    """Section 68's 'logic ships bundled with the model' claim, made real
    for the first time: captures the ACTUAL source code of a @qai.logic
    function and writes it to a file, so it can be reloaded later --
    separately from the model's weights, but genuinely saveable now."""
    import inspect
    source = inspect.getsource(logic_fn.__wrapped__ if hasattr(logic_fn, "__wrapped__") else logic_fn)
    with open(path, "w") as f:
        f.write(source)
    return path


def load_logic_source(path, namespace=None):
    """Reloads a saved logic function's source and executes it in a given
    namespace (which must provide whatever the function references, e.g.
    a trained `model` variable) -- returns the reconstructed, callable
    function, genuinely re-decorated with @qai.logic again.

    NOTE: the saved source is written exactly as a real user would write
    it -- using `@qai.logic`, not bare `@logic` -- so reloading it needs
    the `qai` MODULE itself available in the exec namespace, not just the
    `logic` function alone. Missing this was a real bug, caught immediately
    by actually testing a subprocess reload rather than assuming it worked."""
    import sys
    with open(path) as f:
        source = f.read()
    exec_namespace = dict(namespace or {})
    exec_namespace["qai"] = sys.modules.get("qai") or __import__("qai")
    exec_namespace["logic"] = logic
    exec(source, exec_namespace)
    original_keys = set((namespace or {}).keys()) | {"qai", "logic"}
    new_names = [k for k in exec_namespace if k not in original_keys and callable(exec_namespace[k])]
    if not new_names:
        raise LogicError(f"[WHAT] No function found in the reloaded source at {path}\n"
                          f"[WHY] save_logic_source() expects exactly one function definition\n"
                          f"[FIX] Check the saved file only contains one @qai.logic function")
    return exec_namespace[new_names[0]]
