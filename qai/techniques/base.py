"""
qai.techniques.base — the open Technique/Type interface.

Every built-in technique (Regression, Classifier, ...) is just a normal
Python class implementing this small contract. A custom technique someone
writes themselves does exactly the same thing — nothing about built-ins
is special-cased, matching the design doc's core philosophy.
"""
from __future__ import annotations
import numpy as np


class Technique:
    """Base class every Type extends.

    Required to override: fit(), forward()
    Everything else (help, is_trained, params) is shared for free.
    """

    name: str = "technique"
    family: str = "generic"
    compatible_learning_techniques: tuple = ("supervised",)  # Section 66: which learning_technique(s) this Type actually suits

    def __init__(self, **params):
        self.params = params
        self._trained = False

    # --- required contract ---------------------------------------------
    def fit(self, X, y):
        raise NotImplementedError(f"{self.name} must implement fit()")

    def forward(self, x):
        raise NotImplementedError(f"{self.name} must implement forward()")

    # --- shared, every technique gets these for free --------------------
    def is_trained(self) -> bool:
        return self._trained

    def help(self) -> str:
        status = "trained" if self._trained else "untrained"
        return f"[{self.name}] family={self.family} params={self.params} status={status}"

    def __repr__(self):
        return f"<Technique:{self.name} trained={self._trained}>"
