"""
qai.datasets — real dataset loaders for evaluation.

Provides:
  - load_iris() — Fisher's iris dataset (classification, 150 samples, 4 features)
  - load_digits() — MNIST-like 8x8 digits (classification, 1797 samples, 64 features)
  - load_wine() — wine quality (classification, 178 samples, 13 features)
  - load_breast_cancer() — diagnostic (classification, 569 samples, 30 features)
  - load_diabetes() — regression, 442 samples, 10 features
  - load_california_housing() — regression, 20640 samples, 8 features (optional)
  - make_classification / make_regression / make_blobs — synthetic (already in qai)
  - load_mnist() — MNIST 28x28 (optional, downloads from openml)
  - load_cifar10() — CIFAR-10 32x32 (optional, downloads from torachvision)
  - load_text() — load a text corpus for language modeling
  - load_glue() — GLUE benchmark (optional, requires datasets library)

Usage:
    import qai
    X, y = qai.load_iris()
    X, y = qai.load_digits()
    text = qai.load_text("shakespeare")  # small corpus
"""
from __future__ import annotations
import os, json, urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


_CACHE_DIR = os.path.expanduser("~/.qai/datasets")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _download(url: str, dest: str) -> str:
    """Download a file with caching."""
    if os.path.exists(dest):
        return dest
    print(f"[qai.datasets] downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


# =====================================================================
# Tabular datasets (via sklearn — bundled with qai's requirements)
# =====================================================================
def load_iris() -> Tuple[np.ndarray, np.ndarray]:
    """Fisher's iris dataset (150 samples, 4 features, 3 classes).

    Returns:
        X: (150, 4) float array — sepal/petal length/width
        y: (150,) int array — class labels (0, 1, 2)
    """
    from sklearn.datasets import load_iris as _load
    data = _load()
    return data.data, data.target


def load_digits() -> Tuple[np.ndarray, np.ndarray]:
    """8x8 digits dataset (1797 samples, 64 features, 10 classes).

    Returns:
        X: (1797, 64) float array — 8x8 pixel values (0-16)
        y: (1797,) int array — digit class (0-9)
    """
    from sklearn.datasets import load_digits as _load
    data = _load()
    return data.data, data.target


def load_wine() -> Tuple[np.ndarray, np.ndarray]:
    """Wine dataset (178 samples, 13 features, 3 classes)."""
    from sklearn.datasets import load_wine as _load
    data = _load()
    return data.data, data.target


def load_breast_cancer() -> Tuple[np.ndarray, np.ndarray]:
    """Breast cancer dataset (569 samples, 30 features, 2 classes)."""
    from sklearn.datasets import load_breast_cancer as _load
    data = _load()
    return data.data, data.target


def load_diabetes() -> Tuple[np.ndarray, np.ndarray]:
    """Diabetes regression dataset (442 samples, 10 features)."""
    from sklearn.datasets import load_diabetes as _load
    data = _load()
    return data.data, data.target


def load_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    """California housing regression (20640 samples, 8 features)."""
    from sklearn.datasets import fetch_california_housing as _fetch
    data = _fetch()
    return data.data, data.target


# =====================================================================
# Image datasets (via torchvision — optional)
# =====================================================================
def load_mnist(n_samples: Optional[int] = None, flatten: bool = True
                ) -> Tuple[np.ndarray, np.ndarray]:
    """MNIST 28x28 handwritten digits (70000 samples, 10 classes).

    Args:
        n_samples: if given, return only the first n_samples
        flatten: if True, return (N, 784); if False, return (N, 1, 28, 28)

    Requires: `pip install torchvision`
    """
    try:
        import torchvision
    except ImportError:
        raise ImportError(
            "[WHAT] load_mnist() requires torchvision\n"
            "[FIX] pip install torchvision"
        )
    from torchvision import datasets
    path = os.path.join(_CACHE_DIR, "mnist")
    os.makedirs(path, exist_ok=True)
    train = datasets.MNIST(path, train=True, download=True)
    test = datasets.MNIST(path, train=False, download=True)
    X = np.concatenate([train.data.numpy(), test.data.numpy()]).astype(np.float32) / 255.0
    y = np.concatenate([train.targets.numpy(), test.targets.numpy()]).astype(np.int64)
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    if flatten:
        X = X.reshape(len(X), -1)
    else:
        X = X[:, None, :, :]  # add channel dim
    return X, y


def load_cifar10(n_samples: Optional[int] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """CIFAR-10 32x32 color images (60000 samples, 10 classes).

    Returns:
        X: (N, 3, 32, 32) float array (0-1)
        y: (N,) int array (0-9)

    Requires: `pip install torchvision`
    """
    try:
        import torchvision
    except ImportError:
        raise ImportError(
            "[WHAT] load_cifar10() requires torchvision\n"
            "[FIX] pip install torchvision"
        )
    from torchvision import datasets
    path = os.path.join(_CACHE_DIR, "cifar10")
    os.makedirs(path, exist_ok=True)
    train = datasets.CIFAR10(path, train=True, download=True)
    test = datasets.CIFAR10(path, train=False, download=True)
    X = np.concatenate([train.data, test.data]).astype(np.float32) / 255.0
    y = np.concatenate([np.array(train.targets), np.array(test.targets)]).astype(np.int64)
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    # (N, H, W, C) → (N, C, H, W)
    X = X.transpose(0, 3, 1, 2)
    return X, y


# =====================================================================
# Text datasets
# =====================================================================
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(name: str = "shakespeare", max_chars: Optional[int] = None) -> str:
    """Load a text corpus for language modeling.

    Args:
        name: "shakespeare" (default) — tinyshakespeare (~1MB)
        max_chars: if given, return only the first max_chars

    Returns:
        The text as a string.
    """
    if name == "shakespeare":
        path = os.path.join(_CACHE_DIR, "shakespeare.txt")
        _download(SHAKESPEARE_URL, path)
        with open(path, "r") as f:
            text = f.read()
        if max_chars is not None:
            text = text[:max_chars]
        return text
    raise ValueError(f"Unknown text dataset: {name}. Available: shakespeare")


def load_text_tokens(name: str = "shakespeare", tokenizer=None,
                       max_chars: Optional[int] = None,
                       max_seq_len: int = 64) -> np.ndarray:
    """Load a text corpus and tokenize it for language modeling.

    Args:
        name: dataset name
        tokenizer: a qai tokenizer (BPETokenizer, CharTokenizer, etc.)
                   If None, uses CharTokenizer
        max_chars: limit raw text length
        max_seq_len: split into chunks of this length

    Returns:
        token_ids: (N, max_seq_len) int array
    """
    text = load_text(name, max_chars=max_chars)
    if tokenizer is None:
        from .tokenizers import CharTokenizer
        tokenizer = CharTokenizer()
        tokenizer.train([text])
    # Tokenize
    ids = tokenizer.encode(text)
    if not isinstance(ids, list):
        ids = list(ids)
    # Truncate to multiple of max_seq_len
    n_chunks = len(ids) // max_seq_len
    ids = ids[:n_chunks * max_seq_len]
    arr = np.array(ids, dtype=np.int64).reshape(n_chunks, max_seq_len)
    return arr


# =====================================================================
# GLUE / HuggingFace datasets (optional)
# =====================================================================
def load_glue(task: str = "sst2", split: str = "train") -> Dict:
    """Load a GLUE task via HuggingFace datasets.

    Args:
        task: "sst2", "mrpc", "cola", "qqp", "mnli", "qnli", "rte", "wnli", "stsb"
        split: "train", "validation", "test"

    Returns:
        Dataset dict with 'text' and 'label' fields.

    Requires: `pip install datasets`
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "[WHAT] load_glue() requires the `datasets` library\n"
            "[FIX] pip install datasets"
        )
    return load_dataset("glue", task, split=split)


def load_dataset(name: str, **kwargs) -> Any:
    """Generic HuggingFace datasets loader.

    Args:
        name: dataset name (e.g. "squad", "wikipedia", "imdb")
        **kwargs: passed to datasets.load_dataset

    Requires: `pip install datasets`
    """
    try:
        from datasets import load_dataset as _load
    except ImportError:
        raise ImportError(
            "[WHAT] load_dataset() requires the `datasets` library\n"
            "[FIX] pip install datasets"
        )
    return _load(name, **kwargs)


# =====================================================================
# Synthetic data (re-exports from qai.core.generators for convenience)
# =====================================================================
def make_classification_data(n_samples: int = 100, n_features: int = 4,
                              n_classes: int = 2, random_state: int = 42
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data."""
    from .core.generators import make_classification as _mc
    return _mc(n_samples=n_samples, n_features=n_features,
                n_classes=n_classes, random_state=random_state)


def make_regression_data(n_samples: int = 100, n_features: int = 4,
                          noise: float = 0.1, random_state: int = 42
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    from .core.generators import make_regression as _mr
    return _mr(n_samples=n_samples, n_features=n_features,
                noise=noise, random_state=random_state)


def make_blobs_data(n_samples: int = 100, n_centers: int = 3,
                     n_features: int = 2, random_state: int = 42
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate isotropic Gaussian blobs for clustering."""
    from .core.generators import make_blobs as _mb
    return _mb(n_samples=n_samples, n_centers=n_centers,
                n_features=n_features, random_state=random_state)


__all__ = ["load_iris", "load_digits", "load_wine", "load_breast_cancer",
           "load_diabetes", "load_california_housing",
           "load_mnist", "load_cifar10",
           "load_text", "load_text_tokens",
           "load_glue", "load_dataset",
           "make_classification_data", "make_regression_data", "make_blobs_data"]
