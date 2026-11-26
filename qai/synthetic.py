"""
qai.synthetic -- synthetic data generators for testing + benchmarks.

Beyond make_classification / make_regression (which are toy-grade),
this module provides generators for:
  - make_time_series   : periodic + trend + noise time series
  - make_images        : random images (with optional shapes drawn in)
  - make_text          : random text from a vocab
  - make_graphs        : random graphs (adjacency matrices + node features)

All generators return numpy arrays (or lists of arrays) -- no pandas,
no torch, no external deps.

Usage:
    import qai
    X, y = qai.make_time_series(n_series=100, length=50, n_patterns=3)
    images = qai.make_images(n=100, size=(28, 28), channels=1)
    texts = qai.make_text(n=50, vocab_size=100, length=20)
    graphs = qai.make_graphs(n=20, n_nodes=10, n_features=4)
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


def make_time_series(n_series: int = 100, length: int = 50,
                     n_patterns: int = 3, noise: float = 0.1,
                     random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time series with periodic patterns + trends.

    Each series is a mix of:
      - A sine wave with random frequency + phase
      - A linear trend with random slope
      - Gaussian noise

    Args:
        n_series: number of series to generate.
        length: number of time steps per series.
        n_patterns: number of distinct pattern classes (returned as y).
        noise: noise standard deviation.
        random_state: RNG seed.

    Returns:
        (X, y) where X has shape (n_series, length) and y has shape (n_series,).
        Each y in [0, n_patterns).
    """
    rng = np.random.default_rng(random_state)
    X = np.zeros((n_series, length), dtype=np.float32)
    y = np.zeros(n_series, dtype=np.int64)
    t = np.arange(length, dtype=np.float32)
    # Each pattern class has a characteristic frequency
    pattern_freqs = [0.1, 0.3, 0.5, 0.7, 1.0][:n_patterns]
    for i in range(n_series):
        cls = int(rng.integers(0, n_patterns))
        y[i] = cls
        freq = pattern_freqs[cls] + rng.normal(0, 0.05)
        phase = rng.uniform(0, 2 * np.pi)
        slope = rng.normal(0, 0.01)
        signal = np.sin(2 * np.pi * freq * t + phase) + slope * t
        X[i] = signal + rng.normal(0, noise, length)
    return X, y


def make_images(n: int = 100, size: Tuple[int, int] = (28, 28),
                channels: int = 1, n_classes: int = 3,
                random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images with distinguishable patterns per class.

    Each class has a characteristic shape drawn into the image:
      - class 0: horizontal lines
      - class 1: vertical lines
      - class 2: diagonal lines
      - class 3: blobs (circles)

    Args:
        n: number of images.
        size: (height, width) of each image.
        channels: 1 (grayscale) or 3 (RGB).
        n_classes: number of distinct classes (max 4).
        random_state: RNG seed.

    Returns:
        (X, y) where X has shape (n, channels, H, W) and y has shape (n,).
    """
    rng = np.random.default_rng(random_state)
    H, W = size
    X = np.zeros((n, channels, H, W), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cls = int(rng.integers(0, n_classes))
        y[i] = cls
        img = np.zeros((H, W), dtype=np.float32)
        if cls == 0:  # horizontal lines
            for _ in range(rng.integers(3, 7)):
                row = int(rng.integers(0, H))
                img[row, :] = 1.0
        elif cls == 1:  # vertical lines
            for _ in range(rng.integers(3, 7)):
                col = int(rng.integers(0, W))
                img[:, col] = 1.0
        elif cls == 2:  # diagonal
            for k in range(-W, H):
                if 0 <= k < H + W and rng.random() < 0.3:
                    for r in range(H):
                        c = r - k
                        if 0 <= c < W:
                            img[r, c] = 1.0
        elif cls == 3:  # blob
            cy, cx = int(rng.integers(2, H-2)), int(rng.integers(2, W-2))
            radius = int(rng.integers(1, min(H, W) // 3))
            for r in range(H):
                for c in range(W):
                    if (r - cy)**2 + (c - cx)**2 <= radius**2:
                        img[r, c] = 1.0
        # Add noise
        img += rng.normal(0, 0.1, (H, W))
        img = np.clip(img, 0, 1)
        for ch in range(channels):
            X[i, ch] = img
    return X, y


def make_text(n: int = 50, vocab_size: int = 100, length: int = 20,
              n_classes: int = 3, random_state: Optional[int] = None
              ) -> Tuple[List[List[int]], np.ndarray]:
    """Generate synthetic text (token sequences) with class-specific patterns.

    Each class has a distinct vocabulary subset that appears more frequently.

    Args:
        n: number of documents.
        vocab_size: total vocabulary size.
        length: number of tokens per document.
        n_classes: number of distinct classes.
        random_state: RNG seed.

    Returns:
        (docs, y) where docs is a list of token-id lists and y has shape (n,).
    """
    rng = np.random.default_rng(random_state)
    # Each class gets a preferred subset of the vocabulary
    vocab_per_class = vocab_size // n_classes
    class_vocabs = []
    for c in range(n_classes):
        start = c * vocab_per_class
        end = (c + 1) * vocab_per_class
        class_vocabs.append(list(range(start, end)))
    docs = []
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cls = int(rng.integers(0, n_classes))
        y[i] = cls
        # 70% of tokens from the class's preferred vocab, 30% from anywhere
        doc = []
        for _ in range(length):
            if rng.random() < 0.7:
                doc.append(int(rng.choice(class_vocabs[cls])))
            else:
                doc.append(int(rng.integers(0, vocab_size)))
        docs.append(doc)
    return docs, y


def make_graphs(n: int = 20, n_nodes: int = 10, n_features: int = 4,
                edge_prob: float = 0.3, n_classes: int = 2,
                random_state: Optional[int] = None
                ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Generate synthetic graphs (adjacency matrices + node features).

    Args:
        n: number of graphs.
        n_nodes: number of nodes per graph.
        n_features: number of features per node.
        edge_prob: probability of an edge between any two nodes.
        n_classes: number of distinct graph classes.
        random_state: RNG seed.

    Returns:
        (adjacency, features, y) where:
          - adjacency: list of (n_nodes, n_nodes) numpy arrays
          - features: list of (n_nodes, n_features) numpy arrays
          - y: shape (n,) class labels
    """
    rng = np.random.default_rng(random_state)
    adjacency = []
    features = []
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cls = int(rng.integers(0, n_classes))
        y[i] = cls
        # Class-specific edge density
        density = edge_prob * (1.5 if cls == 0 else 0.5)
        adj = (rng.random((n_nodes, n_nodes)) < density).astype(np.float32)
        # Make symmetric (undirected)
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)  # no self-loops
        adjacency.append(adj)
        # Node features: class-specific bias
        feat = rng.standard_normal((n_nodes, n_features)).astype(np.float32)
        feat += cls  # shift features by class
        features.append(feat)
    return adjacency, features, y


__all__ = ["make_time_series", "make_images", "make_text", "make_graphs"]
