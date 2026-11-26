"""
qai.core.generators -- Synthetic dataset generator utilities.

Provides make_classification() and make_regression() with rich parameter
sets matching the common sklearn signatures -- so you can actually
benchmark robustness to label noise, redundant features, class
separation, etc., instead of being stuck with one toy shape.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def make_classification(
    n_samples: int = 100,
    n_features: int = 4,
    n_informative: Optional[int] = None,
    n_redundant: int = 0,
    n_repeated: int = 0,
    n_classes: int = 2,
    n_clusters_per_class: int = 2,
    flip_y: float = 0.0,
    class_sep: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic classification dataset with full control over
    feature informativeness, label noise, and class separation.

    Args:
        n_samples: number of samples.
        n_features: total features (informative + redundant + repeated + noise).
        n_informative: number of informative features (default: n_features - n_redundant - n_repeated).
        n_redundant: number of redundant features (linear combos of informative).
        n_repeated: number of duplicated features (copies of informative).
        n_classes: number of classes.
        n_clusters_per_class: clusters per class.
        flip_y: fraction of labels to randomly flip (label noise).
        class_sep: class separation (larger = easier).
        random_state: RNG seed.

    Returns:
        (X, y) -- X shape (n_samples, n_features), y shape (n_samples,).
    """
    if n_informative is None:
        n_informative = max(2, n_features - n_redundant - n_repeated)
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError(
            f"[WHAT] n_informative + n_redundant + n_repeated > n_features\n"
            f"[WHY] {n_informative} + {n_redundant} + {n_repeated} > {n_features}\n"
            f"[FIX] Increase n_features or reduce one of the others"
        )
    from sklearn.datasets import make_classification as sk_make
    X, y = sk_make(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=random_state,
    )
    return X, y


def make_regression(
    n_samples: int = 100,
    n_features: int = 4,
    n_informative: Optional[int] = None,
    n_targets: int = 1,
    noise: float = 0.1,
    bias: float = 0.0,
    coef: bool = False,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset.

    Args:
        n_samples: number of samples.
        n_features: total features.
        n_informative: number of informative features (default: n_features).
        n_targets: number of target variables.
        noise: std dev of Gaussian noise added to the target.
        bias: bias term added to the target.
        coef: if True, also return the true coefficients.
        random_state: RNG seed.

    Returns:
        (X, y) or (X, y, coef) if coef=True.
    """
    if n_informative is None:
        n_informative = n_features
    from sklearn.datasets import make_regression as sk_make_reg
    result = sk_make_reg(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        noise=noise,
        bias=bias,
        coef=coef,
        random_state=random_state,
    )
    return result


def make_blobs(
    n_samples: int = 100,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate isotropic Gaussian blobs for clustering tests.

    Args:
        n_samples: total samples (split equally across centers if int).
        n_features: number of features.
        centers: number of centers (int) or explicit center coords.
        cluster_std: std dev of clusters.
        center_box: bounding box for random center generation.
        random_state: RNG seed.

    Returns:
        (X, y) -- X shape (n_samples, n_features), y has cluster labels.
    """
    from sklearn.datasets import make_blobs as sk_make_blobs
    X, y = sk_make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        center_box=center_box,
        random_state=random_state,
    )
    return X, y
