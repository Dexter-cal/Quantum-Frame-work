"""
qai.learning -- the LearningTechnique registry (Section 66/67), a separate
axis from qai.techniques (which is architecture/Type).
"""

from __future__ import annotations
from .base import LearningTechnique
from .supervised import (
    Supervised, Unsupervised,
    MultiOutputClassifier, MultiOutputRegressor,
    ClassifierChain, StackingClassifier, StackingRegressor,
)
from .reinforcement import Reinforcement

REGISTRY = {
    "supervised": Supervised,
    "unsupervised": Unsupervised,
    "reinforcement": Reinforcement,
}


def get_learning_technique(name: str):
    if name not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise ValueError(f"Unknown learning_technique '{name}'. Available: {available}")
    return REGISTRY[name]
