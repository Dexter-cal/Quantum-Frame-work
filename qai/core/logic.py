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


def logic(fn=None, *, name=None, max_seconds=None, version="1.0.0",
           description="", tags=None, author=None, retries=0,
           cache=False, validate_input=None, validate_output=None,
           requires=None, source=None):
    """Decorator implementing the logic{} contract.

    Usage:
        @qai.logic
        def my_logic(x):
            ...
            return result

        @qai.logic(name="retry_until_confident", max_seconds=5,
                    version="2.0.0", description="Retries until confident",
                    tags=["classification", "retry"], author="alice",
                    retries=3, cache=True,
                    requires=["agent", "tokenizer"],
                    source="def my_logic(x):\\n    return agent.predict(x)")
        def my_logic(x):
            ...
            return result

    Args:
        name:            Custom name for the logic block (default: function name)
        max_seconds:     Hard time budget; raises LogicError if exceeded
        version:         Semantic version string (stored in metadata)
        description:     Human-readable description
        tags:            List of string tags for search/categorisation
        author:          Author name
        retries:         Number of times to retry on exception (0 = no retry)
        cache:           If True, memoize results (same inputs → same output)
        validate_input:  Optional callable(input) -> bool; raises if False
        validate_output: Optional callable(output) -> bool; raises if False
        requires:        List of names this function expects in its exec
                         namespace at bundle-load time. The bundle loader
                         verifies all `requires` names are present and
                         raises a clear error if not. Default: None.
                         Example: requires=["agent"] for a function that
                         references a global `agent` variable.
        source:          Optional explicit source code string. By default
                         the decorator uses inspect.getsource(fn), which
                         fails in REPLs and Jupyter (functions defined in
                         exec() have no source file). Set `source=` to
                         ship the source manually -- useful for notebook
                         workflows.
    """
    tags = tags or []

    def decorator(func):
        logic_name = name or func.__name__
        _cache: dict = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # requires= enforcement: check that every name in `requires`
            # is resolvable in the function's globals (so missing deps
            # raise a clear LogicError instead of a confusing NameError
            # deep inside the call).
            # Check BOTH func.__globals__ (the original function's module)
            # AND wrapper.__globals__ (the decorator's module, where users
            # typically inject `agent` after load_bundle).
            if requires:
                combined_globals = {**func.__globals__, **wrapper.__globals__}
                missing = [n for n in requires
                           if n not in combined_globals]
                if missing:
                    raise LogicError(
                        f"[WHAT] logic '{logic_name}' is missing required namespace deps: {missing}\n"
                        f"[WHY] The function declared requires={requires!r} but the bundle "
                        f"loader did not inject these names into the exec namespace.\n"
                        f"[FIX] After load_bundle(), inject the missing names via "
                        f"fn.__globals__['{missing[0]}'] = <your object>, or use "
                        f"@qai.logic(..., requires=None) to skip this check."
                    )

            # input validation
            if validate_input is not None:
                try:
                    if not validate_input(args[0] if args else kwargs):
                        raise LogicError(
                            f"[WHAT] logic '{logic_name}' input validation failed\n"
                            f"[WHY] validate_input returned False\n"
                            f"[FIX] Check the input contract of '{logic_name}'"
                        )
                except LogicError:
                    raise
                except Exception as e:
                    raise LogicError(f"validate_input raised: {e}")

            # caching
            if cache:
                cache_key = (args, tuple(sorted(kwargs.items())))
                if cache_key in _cache:
                    return _cache[cache_key]

            start = time.time()
            last_error = None

            # retry loop
            for attempt in range(retries + 1):
                try:
                    result = func(*args, **kwargs)
                    break
                except Exception as e:
                    last_error = e
                    if attempt < retries:
                        time.sleep(0.1 * (attempt + 1))  # backoff
                        continue
                    raise

            elapsed = time.time() - start

            # ENFORCE: output() is mandatory
            if result is None:
                raise LogicError(
                    f"[WHAT] logic function '{logic_name}' completed without returning a value\n"
                    f"[WHY] Every logic block must have an explicit output -- returning None "
                    f"is treated as a missing exit, not a valid (empty) result\n"
                    f"[FIX] Add an explicit 'return <value>' at the end of '{logic_name}'"
                )

            # output validation
            if validate_output is not None:
                try:
                    if not validate_output(result):
                        raise LogicError(
                            f"[WHAT] logic '{logic_name}' output validation failed\n"
                            f"[WHY] validate_output returned False\n"
                            f"[FIX] Check the output contract of '{logic_name}'"
                        )
                except LogicError:
                    raise
                except Exception as e:
                    raise LogicError(f"validate_output raised: {e}")

            # time budget
            if max_seconds is not None and elapsed > max_seconds:
                raise LogicError(
                    f"[WHAT] logic function '{logic_name}' took {elapsed:.2f}s\n"
                    f"[WHY] max_seconds={max_seconds} was set as a hard time budget\n"
                    f"[FIX] Either optimize '{logic_name}' or raise max_seconds if this is expected"
                )

            # cache store
            if cache:
                _cache[cache_key] = result

            wrapper.last_run = {
                "logic_name": logic_name,
                "elapsed_s": round(elapsed, 4),
                "result_type": type(result).__name__,
                "version": version,
                "attempts": retries + 1,
            }
            return result

        wrapper.logic_name = logic_name
        wrapper.last_run = None
        wrapper.logic_metadata = {
            "name": logic_name,
            "version": version,
            "description": description,
            "tags": tags,
            "author": author,
            "max_seconds": max_seconds,
            "retries": retries,
            "cache": cache,
            "requires": list(requires) if requires else [],
        }
        # Source: explicit > dill > inspect.getsource (which fails in REPLs)
        # Stored as `__logic_source__` so bundle capture can fall back to it
        if source is not None:
            wrapper.__logic_source__ = source
        else:
            # Try inspect.getsource first (works for .py files)
            try:
                import inspect as _inspect
                wrapper.__logic_source__ = _inspect.getsource(func)
            except (OSError, TypeError):
                # Function defined in REPL/Jupyter/exec -- try dill
                try:
                    import dill as _dill
                    wrapper.__logic_source__ = _dill.source.getsource(func)
                except (ImportError, Exception):
                    # No dill available -- source is None
                    wrapper.__logic_source__ = None
        return wrapper

    if fn is not None and callable(fn):
        # used as @qai.logic with no parens
        return decorator(fn)
    return decorator


def save_logic_source(logic_fn, path, auto_capture=True):
    """Section 68's 'logic ships bundled with the model' claim, made real:
    captures the ACTUAL source code of a @qai.logic function and writes it
    to a file, so it can be reloaded later -- separately from the model's
    weights, but genuinely saveable now.

    Args:
        logic_fn: The decorated function (output of @qai.logic).
        path: Where to write the source file.
        auto_capture: If True (default), append a `# closure deps:` comment
            listing the names the function closes over, so callers know what
            to pass to load_logic_source(namespace=...). This is a hint --
            actual closure values still need to be supplied by the caller at
            reload time (we can't pickle arbitrary Python objects into a
            .py file).
    """
    import inspect
    target = logic_fn.__wrapped__ if hasattr(logic_fn, "__wrapped__") else logic_fn
    # Source: prefer explicit __logic_source__ (set via @qai.logic(source=...))
    # so REPL/Jupyter-defined functions can still be saved.
    source = getattr(logic_fn, "__logic_source__", None)
    if source is None:
        try:
            source = inspect.getsource(target)
        except (OSError, TypeError) as e:
            raise LogicError(
                f"[WHAT] Could not capture source for logic function "
                f"'{getattr(logic_fn, 'logic_name', logic_fn.__name__)}'\n"
                f"[WHY] inspect.getsource() failed: {e}. This happens for "
                f"functions defined in REPLs/Jupyter/exec.\n"
                f"[FIX] Re-decorate with @qai.logic(source='''...your code...''') "
                f"to ship the source explicitly."
            )

    closure_deps = []
    if auto_capture:
        try:
            cv = inspect.getclosurevars(target)
            # Names from enclosing scope (nonlocals) + globals the fn actually uses
            closure_deps = sorted(set(cv.nonlocals.keys()) | set(cv.globals.keys()))
        except (TypeError, NameError):
            # getclosurevars fails on builtins or C-implemented functions
            pass

    # Include the `requires` list from logic_metadata, if present
    requires = (getattr(logic_fn, "logic_metadata", {}) or {}).get("requires", [])
    if requires:
        closure_deps = sorted(set(closure_deps) | set(requires))

    with open(path, "w") as f:
        f.write(source)
        if closure_deps:
            f.write(f"\n# closure deps (pass these to load_logic_source(namespace=...)): "
                    f"{closure_deps}\n")
        if requires:
            f.write(f"# required namespace names: {requires}\n")
    return path


def load_logic_source(path, namespace=None):
    """Reloads a saved logic function's source and executes it in a given
    namespace (which must provide whatever the function references, e.g.
    a trained `model` variable) -- returns the reconstructed, callable
    function, genuinely re-decorated with @qai.logic again.

    The saved source uses `@qai.logic` syntax, so the `qai` MODULE must be
    available in the exec namespace. We add it automatically. If the saved
    file has a `# closure deps:` comment (from auto_capture=True), callers
    must supply those names in `namespace=` -- otherwise the reloaded
    function will NameError on first call.
    """
    import sys
    with open(path) as f:
        source = f.read()
    exec_namespace = dict(namespace or {})
    exec_namespace["qai"] = sys.modules.get("qai") or __import__("qai")
    exec_namespace["logic"] = logic
    exec(source, exec_namespace)
    original_keys = set((namespace or {}).keys()) | {"qai", "logic", "__builtins__"}
    new_names = [k for k in exec_namespace
                 if k not in original_keys and callable(exec_namespace[k])]
    if not new_names:
        raise LogicError(f"[WHAT] No function found in the reloaded source at {path}\n"
                          f"[WHY] save_logic_source() expects exactly one function definition\n"
                          f"[FIX] Check the saved file only contains one @qai.logic function")
    return exec_namespace[new_names[0]]
