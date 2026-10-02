"""
qai.core.generators -- Synthetic dataset generator utilities.
Provides make_classification() for generating customizable synthetic datasets.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np


def make_classification(n_samples: int = 100, n_features: int = 4, n_classes: int = 2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic classification dataset."""
    from sklearn.datasets import make_classification as sk_make
    X, y = sk_make(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 1),
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state
    )
    return X, y


def make_regression(n_samples: int = 100, n_features: int = 4, noise: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic regression dataset."""
    from sklearn.datasets import make_regression as sk_make_reg
    X, y = sk_make_reg(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    return X, y
