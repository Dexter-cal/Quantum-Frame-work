"""
qai.techniques.base — the open Technique/Type interface.

Every built-in technique (Regression, Classifier, ...) is just a normal
Python class implementing this small contract. A custom technique someone
writes themselves does exactly the same thing — nothing about built-ins
is special-cased, matching the design doc's core philosophy.
"""
from __future__ import annotations
import inspect
import numpy as np


class Technique:
    """Base class every Type extends.

    Required to override: fit(), forward()
    Everything else (help, is_trained, params) is shared for free.

    Param introspection (sklearn-style):
      - get_params(): returns dict of all constructor params
      - set_params(**kwargs): update params (like sklearn)
      - list_params(): returns list of param names with types + defaults
      - param_help(): returns human-readable param documentation
      - validate_params(**kwargs): check if params are valid
    """

    name: str = "technique"
    family: str = "generic"
    compatible_learning_techniques: tuple = ("supervised",)

    def __init__(self, **params):
        self.params = params
        self._trained = False

    # --- required contract ---------------------------------------------
    def fit(self, X, y):
        raise NotImplementedError(f"{self.name} must implement fit()")

    def train(self, X, y=None):
        """Alias for fit(). Some users expect model.train(); we provide it."""
        return self.fit(X, y)

    def forward(self, x):
        raise NotImplementedError(f"{self.name} must implement forward()")

    def eval(self):
        """Switch all torch.nn.Module sub-components to eval mode."""
        try:
            import torch.nn as nn
            for attr in vars(self).values():
                if isinstance(attr, nn.Module):
                    attr.eval()
                elif isinstance(attr, (list, tuple)):
                    for x in attr:
                        if isinstance(x, nn.Module):
                            x.eval()
        except Exception:
            pass
        return self

    def train_mode(self):
        """Switch torch sub-components back to train mode."""
        try:
            import torch.nn as nn
            for attr in vars(self).values():
                if isinstance(attr, nn.Module):
                    attr.train()
                elif isinstance(attr, (list, tuple)):
                    for x in attr:
                        if isinstance(x, nn.Module):
                            x.train()
        except Exception:
            pass
        return self

    # --- shared, every technique gets these for free --------------------
    def is_trained(self) -> bool:
        return self._trained

    # --- param introspection (sklearn-style) ---------------------------
    def get_params(self) -> dict:
        """Get all constructor parameters of this technique.

        Returns a dict of param_name → current_value, matching sklearn's API.

        Example:
            >>> m = qai.build(technique="transformer", d_model=256)
            >>> m.technique.get_params()
            {'vocab_size': 256, 'd_model': 256, 'n_heads': 4, 'n_layers': 2, ...}
        """
        try:
            sig = inspect.signature(self.__class__.__init__)
            params = {}
            for name, param in sig.parameters.items():
                if name in ("self", "params", "kw", "kwargs"):
                    continue
                # Get the value from the instance, or the default
                if hasattr(self, name):
                    params[name] = getattr(self, name)
                elif param.default is not inspect.Parameter.empty:
                    params[name] = param.default
            return params
        except (ValueError, TypeError):
            return dict(self.params) if self.params else {}

    def set_params(self, **kwargs) -> "Technique":
        """Update parameters on this technique (like sklearn's set_params).

        Note: this updates the stored values but does NOT re-initialize the
        model. For changes to take effect, you typically need to rebuild:
            m = qai.build(technique="transformer", **m.technique.get_params())

        Example:
            >>> m.technique.set_params(d_model=512, n_heads=8)
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            self.params[k] = v
        return self

    def list_params(self) -> list:
        """List all available parameters with their types, defaults, and current values.

        Returns:
            List of dicts: [{"name": str, "type": str, "default": Any,
                              "current": Any, "description": str}]
        """
        try:
            sig = inspect.signature(self.__class__.__init__)
            params = []
            for name, param in sig.parameters.items():
                if name in ("self", "params", "kw", "kwargs"):
                    continue
                default = param.default if param.default is not inspect.Parameter.empty else None
                current = getattr(self, name, default)
                # Infer type
                if current is not None:
                    ptype = type(current).__name__
                elif default is not None:
                    ptype = type(default).__name__
                else:
                    ptype = "Any"
                # Get description from docstring (if available)
                desc = ""
                doc = self.__class__.__init__.__doc__ or self.__class__.__doc__ or ""
                if name in doc:
                    # Try to extract the description after the param name
                    for line in doc.split("\n"):
                        if name in line:
                            desc = line.split(name, 1)[-1].strip(": ").strip()
                            break
                params.append({
                    "name": name,
                    "type": ptype,
                    "default": default,
                    "current": current,
                    "description": desc,
                })
            return params
        except (ValueError, TypeError):
            return []

    def param_help(self) -> str:
        """Human-readable parameter documentation.

        Example:
            >>> print(m.technique.param_help())
            transformer parameters:
              vocab_size (int) = 256     # vocabulary size
              d_model (int) = 64         # model dimension
              n_heads (int) = 4          # attention heads
              ...
        """
        params = self.list_params()
        if not params:
            return f"{self.name}: no introspectable parameters"
        lines = [f"{self.name} parameters:"]
        for p in params:
            default_str = repr(p["default"]) if p["default"] is not None else "required"
            current_str = repr(p["current"]) if p["current"] is not None else "None"
            desc = f"  # {p['description']}" if p["description"] else ""
            lines.append(f"  {p['name']} ({p['type']}) = {default_str}{desc}")
            if p["current"] != p["default"] and p["default"] is not None:
                lines.append(f"    → current: {current_str}")
        return "\n".join(lines)

    def validate_params(self, **kwargs) -> tuple:
        """Validate that the given params are accepted by this technique.

        Returns:
            (valid: bool, errors: list[str])
        """
        try:
            sig = inspect.signature(self.__class__.__init__)
            accepted = set()
            has_var_kw = False
            for name, param in sig.parameters.items():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    has_var_kw = True
                else:
                    accepted.add(name)
            errors = []
            for k in kwargs:
                if k not in accepted and not has_var_kw:
                    errors.append(f"Unknown parameter: '{k}'. Accepted: {sorted(accepted)}")
            return (len(errors) == 0, errors)
        except (ValueError, TypeError):
            return (True, [])

    def help(self) -> str:
        """Full help text including params, family, and status."""
        status = "trained" if self._trained else "untrained"
        lines = [
            f"Technique: {self.name}",
            f"  Family: {self.family}",
            f"  Status: {status}",
            f"  Compatible learning: {self.compatible_learning_techniques}",
            "",
            self.param_help(),
        ]
        return "\n".join(lines)

    def __repr__(self):
        trained = "trained" if self._trained else "untrained"
        n_params = len(self.list_params())
        return f"<Technique:{self.name} family={self.family} {trained} params={n_params}>"
